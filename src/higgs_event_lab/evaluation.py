"""Standardized evaluation metrics and threshold analysis for HIGGS classifiers.

Model owners use this module to compute consistent, reproducible validation and
held-out test metrics matching the higgs-experiment-v1 artifact contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute ROC-AUC, PR-AUC, Brier score, and calibration curves.

    Parameters
    ----------
    y_true : 1D array of binary ground truth labels (0 or 1).
    probabilities : 1D array of predicted class-1 probabilities in [0, 1].
    n_bins : Number of bins for uniform calibration curve calculation.
    """
    y = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)

    if y.ndim != 1 or probs.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional arrays")
    if y.shape[0] != probs.shape[0]:
        raise ValueError("y_true and probabilities must have the same length")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("metrics require both binary classes (0 and 1) in the evaluation set")
    if not (np.isfinite(probs).all() and (probs >= 0.0).all() and (probs <= 1.0).all()):
        raise ValueError("probabilities must be finite numbers within [0.0, 1.0]")

    fraction_positives, mean_predicted = calibration_curve(
        y, probs, n_bins=n_bins, strategy="uniform"
    )

    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "brier_score": float(brier_score_loss(y, probs)),
        "calibration": {
            "fraction_positives": [float(val) for val in fraction_positives],
            "mean_predicted_value": [float(val) for val in mean_predicted],
        },
    }


def select_operating_threshold(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    target_signal_efficiency: float = 0.50,
) -> float:
    """Select a decision threshold on validation data to achieve a target signal efficiency.

    The selected threshold is frozen and subsequently evaluated on the held-out test
    set to ensure strictly leakage-free threshold selection.
    """
    if not 0 < target_signal_efficiency < 1:
        raise ValueError("target_signal_efficiency must be between 0 and 1")
    signal_probs = probs_val[y_val == 1]
    if len(signal_probs) == 0:
        raise ValueError("validation set must contain positive signal events")
    threshold = float(np.percentile(signal_probs, (1.0 - target_signal_efficiency) * 100))
    return threshold
