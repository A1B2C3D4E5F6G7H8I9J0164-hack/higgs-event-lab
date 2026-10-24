"""Unit tests for threshold operating point and evaluation curve calculations.

Tests:
- Confusion matrix counts (TP, FP, TN, FN)
- Precision, recall (signal efficiency), background rejection, FPR
- Safe edge-case handling for threshold 0.0, 1.0, and empty selections
- Verification of ROC, PR, and calibration curves against scikit-learn
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from app.metrics import (
    calculate_operating_metrics,
    compute_calibration_points,
    compute_pr_points,
    compute_roc_points,
)


@pytest.fixture
def sample_labels_and_scores():
    """Known binary labels and scores for deterministic metric verification."""
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    # Scores: high for first 3 positives, low for 1 positive (FN at 0.5)
    # Low for first 3 negatives, high for 1 negative (FP at 0.5)
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1], dtype=np.float64)
    return y, scores


def test_calculate_operating_metrics_standard_threshold(sample_labels_and_scores):
    """Verify operating metrics at cutoff 0.50."""
    y, scores = sample_labels_and_scores
    res = calculate_operating_metrics(y, scores, threshold=0.50)

    # Predictions >= 0.50: indices [0, 1, 2, 4]
    # Indices 0, 1, 2 are y=1 (TP=3)
    # Index 4 is y=0 (FP=1)
    # Predictions < 0.50: indices [3, 5, 6, 7]
    # Index 3 is y=1 (FN=1)
    # Indices 5, 6, 7 are y=0 (TN=3)

    assert res["true_positives"] == 3
    assert res["false_positives"] == 1
    assert res["true_negatives"] == 3
    assert res["false_negatives"] == 1

    assert res["total_events"] == 8
    assert res["total_signal"] == 4
    assert res["total_background"] == 4

    assert res["signal_selections"] == 4
    assert res["selection_rate"] == 0.50

    # Precision: TP / (TP + FP) = 3 / 4 = 0.75
    assert res["precision"] == 0.75
    assert not res["precision_undefined"]

    # Recall (Signal Efficiency): TP / (TP + FN) = 3 / 4 = 0.75
    assert res["recall"] == 0.75
    assert res["signal_efficiency"] == 0.75

    # Background Rejection: TN / (TN + FP) = 3 / 4 = 0.75
    assert res["background_rejection"] == 0.75
    assert res["false_positive_rate"] == 0.25

    # Accuracy: (3 + 3) / 8 = 0.75
    assert res["accuracy"] == 0.75


def test_calculate_operating_metrics_edge_case_threshold_zero(sample_labels_and_scores):
    """Threshold 0.0 should select all events as signal."""
    y, scores = sample_labels_and_scores
    res = calculate_operating_metrics(y, scores, threshold=0.0)

    assert res["true_positives"] == 4
    assert res["false_positives"] == 4
    assert res["true_negatives"] == 0
    assert res["false_negatives"] == 0

    assert res["signal_selections"] == 8
    assert res["selection_rate"] == 1.0
    assert res["recall"] == 1.0
    assert res["background_rejection"] == 0.0
    assert res["precision"] == 0.50
    assert not res["precision_undefined"]


def test_calculate_operating_metrics_edge_case_threshold_one(sample_labels_and_scores):
    """Threshold 1.0 with max score < 1.0 should select 0 events without dividing by zero."""
    y, scores = sample_labels_and_scores
    res = calculate_operating_metrics(y, scores, threshold=1.0)

    assert res["true_positives"] == 0
    assert res["false_positives"] == 0
    assert res["true_negatives"] == 4
    assert res["false_negatives"] == 4

    assert res["signal_selections"] == 0
    assert res["selection_rate"] == 0.0
    assert res["recall"] == 0.0
    assert res["background_rejection"] == 1.0

    # Precision is 0/0; verify safe handling
    assert res["precision"] == 0.0
    assert res["precision_undefined"] is True


def test_calculate_operating_metrics_invalid_inputs():
    """Verify exceptions on invalid inputs."""
    y = np.array([1, 0, 1])
    scores = np.array([0.5, 0.4, 0.8])

    # Threshold out of bounds
    with pytest.raises(ValueError, match="threshold must be in"):
        calculate_operating_metrics(y, scores, threshold=-0.1)

    with pytest.raises(ValueError, match="threshold must be in"):
        calculate_operating_metrics(y, scores, threshold=1.05)

    # Empty inputs
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_operating_metrics(np.array([]), np.array([]), threshold=0.5)

    # Non-binary labels
    with pytest.raises(ValueError, match="only contain binary values"):
        calculate_operating_metrics(np.array([1, 2, 0]), scores, threshold=0.5)

    # Non-finite scores
    with pytest.raises(ValueError, match="finite numbers"):
        calculate_operating_metrics(y, np.array([0.5, np.nan, 0.8]), threshold=0.5)

    # Scores out of [0, 1]
    with pytest.raises(ValueError, match="bounded within"):
        calculate_operating_metrics(y, np.array([0.5, 1.2, 0.8]), threshold=0.5)


def test_compute_roc_points():
    """Verify computed ROC curve points and AUC."""
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=100)
    scores = rng.uniform(0.0, 1.0, size=100)

    fpr, tpr, auc_val = compute_roc_points(y, scores)

    assert len(fpr) == len(tpr)
    assert 0.0 <= auc_val <= 1.0
    assert np.isclose(auc_val, roc_auc_score(y, scores))
    # Monotonicity checks
    assert np.all(np.diff(fpr) >= 0.0)


def test_compute_pr_points():
    """Verify computed Precision-Recall curve points and Average Precision."""
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=100)
    scores = rng.uniform(0.0, 1.0, size=100)

    precision, recall, pr_auc = compute_pr_points(y, scores)

    assert len(precision) == len(recall)
    assert 0.0 <= pr_auc <= 1.0
    assert np.isclose(pr_auc, average_precision_score(y, scores))


def test_compute_calibration_points():
    """Verify reliability diagram calculation and Brier score."""
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=100)
    scores = rng.uniform(0.0, 1.0, size=100)

    frac_pos, mean_pred, brier = compute_calibration_points(y, scores, n_bins=10)

    assert len(frac_pos) == len(mean_pred)
    assert len(frac_pos) <= 10
    assert 0.0 <= brier <= 1.0
    assert np.isclose(brier, brier_score_loss(y, scores))
