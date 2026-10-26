"""Unit tests for Logistic Regression baseline module (MID-03).

Tests training, feature selection, metric integration, artifact export,
leakage protection, and failure modes.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from higgs_event_lab.classical import (
    LogisticRegressionConfig,
    build_pipeline,
    main,
    run_feature_group_benchmarks,
    train_logistic_regression,
    write_run_artifacts,
)
from higgs_event_lab.features import HIGH_LEVEL_FEATURES, LOW_LEVEL_FEATURES
from higgs_event_lab.protocol import (
    ARTIFACT_SCHEMA_VERSION,
    SplitConfig,
    create_stratified_splits,
)


@pytest.fixture
def synthetic_features_and_labels():
    """Deterministic synthetic dataset with 120 samples and 28 features."""
    rng = np.random.default_rng(42)
    n_rows = 120
    X = rng.normal(loc=0.0, scale=1.0, size=(n_rows, 28))
    # Create non-trivial relationship
    score = X[:, 0] * 1.5 - X[:, 21] * 1.0 + rng.normal(0, 0.5, size=n_rows)
    y = (score > np.median(score)).astype(np.int64)
    feature_names = [f"f_{i}" for i in range(28)]
    df_X = pd.DataFrame(X, columns=feature_names)
    return df_X, y


def test_logistic_regression_config_validation():
    """Verify bounds and validation in LogisticRegressionConfig."""
    with pytest.raises(ValueError, match="C must be strictly positive"):
        LogisticRegressionConfig(C=-0.5)

    with pytest.raises(ValueError, match="C must be strictly positive"):
        LogisticRegressionConfig(C=0.0)

    with pytest.raises(ValueError, match="max_iter must be strictly positive"):
        LogisticRegressionConfig(max_iter=0)

    with pytest.raises(ValueError, match="target_signal_efficiency"):
        LogisticRegressionConfig(target_signal_efficiency=1.5)

    with pytest.raises(ValueError, match="tol must be strictly positive"):
        LogisticRegressionConfig(tol=-1e-3)


def test_pipeline_leakage_free_scaling(synthetic_features_and_labels):
    """Verify that StandardScaler fits strictly on train rows via Pipeline."""
    X, y = synthetic_features_and_labels
    split = create_stratified_splits(y, SplitConfig(seed=42))

    config = LogisticRegressionConfig(seed=42, max_iter=200)
    result = train_logistic_regression(X, y, split=split, config=config)

    scaler = result["scaler"]
    train_data = X.to_numpy()[split.train]
    expected_mean = np.mean(train_data, axis=0)

    # Scaler mean must match training partition mean, NOT full dataset mean
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5)
    full_mean = np.mean(X.to_numpy(), axis=0)
    assert not np.allclose(scaler.mean_, full_mean, atol=1e-3)


def test_threshold_selected_on_validation_only(synthetic_features_and_labels):
    """Verify operating threshold is determined from validation predictions."""
    X, y = synthetic_features_and_labels
    split = create_stratified_splits(y, SplitConfig(seed=42))

    config = LogisticRegressionConfig(seed=42, target_signal_efficiency=0.60)
    result = train_logistic_regression(X, y, split=split, config=config)

    threshold = result["operating_threshold"]
    val_probs = result["validation"]["probabilities"]
    val_labels = result["validation"]["labels"]

    val_signal_probs = val_probs[val_labels == 1]
    expected_threshold = float(np.percentile(val_signal_probs, (1.0 - 0.60) * 100))
    assert abs(threshold - expected_threshold) < 1e-6

    # Verify operating point metrics are calculated
    op = result["operating_point"]
    assert 0.0 <= op["validation_actual_signal_efficiency"] <= 1.0
    assert 0.0 <= op["test_actual_signal_efficiency"] <= 1.0
    assert 0.0 <= op["test_background_rejection"] <= 1.0


def test_metric_integration_and_artifact_export(synthetic_features_and_labels, tmp_path: Path):
    """Verify artifact format matches higgs-experiment-v1 contract."""
    X, y = synthetic_features_and_labels
    split = create_stratified_splits(y, SplitConfig(seed=42))
    config = LogisticRegressionConfig(seed=42, max_iter=200)

    result = train_logistic_regression(X, y, split=split, config=config)

    run_dir = tmp_path / "logreg_test_run"
    meta_path, pred_path = write_run_artifacts(
        result,
        run_dir=run_dir,
        run_id="logreg_test_run",
        feature_group="all",
        feature_names=list(X.columns),
        data_source="synthetic_data",
        row_count=len(X),
        config=config,
    )

    assert meta_path.exists()
    assert pred_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert meta["run_id"] == "logreg_test_run"
    assert meta["model"]["name"] == "logistic_regression"
    assert meta["model"]["family"] == "classical_linear"
    assert meta["features"]["group"] == "all"
    assert meta["features"]["count"] == 28

    # Evaluation metrics
    for part in ("validation", "test"):
        metrics = meta["evaluation"][part]
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "brier_score" in metrics
        assert "calibration" in metrics
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    # Predictions CSV
    df_preds = pd.read_csv(pred_path)
    assert list(df_preds.columns) == ["row_index", "class_label", "score"]
    assert len(df_preds) == len(split.test)
    assert (df_preds["score"] >= 0.0).all() and (df_preds["score"] <= 1.0).all()


def test_failure_cases(synthetic_features_and_labels):
    """Verify failure modes on invalid inputs."""
    X, y = synthetic_features_and_labels

    # 1. 1D X array
    with pytest.raises(ValueError, match="two-dimensional"):
        train_logistic_regression(np.zeros(20), y[:20])

    # 2. X with non-finite values
    X_nan = X.copy()
    X_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        train_logistic_regression(X_nan, y)

    # 3. Mismatched length
    with pytest.raises(ValueError, match="matching row counts"):
        train_logistic_regression(X, y[:50])

    # 4. Single class label
    with pytest.raises(ValueError, match="both binary classes"):
        train_logistic_regression(X[:20], np.ones(20, dtype=int))


def test_run_feature_group_benchmarks(tmp_path: Path):
    """Verify multi-group benchmarking across low_level and high_level."""
    data_file = tmp_path / "mock_higgs.csv.gz"
    rng = np.random.default_rng(99)
    labels = rng.integers(0, 2, size=(80, 1)).astype(np.float64)
    features = rng.normal(0.0, 1.0, size=(80, 28)).astype(np.float64)
    matrix = np.hstack([labels, features])

    with gzip.open(data_file, "wt", encoding="utf-8") as f:
        for row in matrix:
            f.write(",".join(f"{val:.6f}" for val in row) + "\n")

    out_dir = tmp_path / "benchmarks"
    results = run_feature_group_benchmarks(
        data_path=data_file,
        row_count=60,
        output_base_dir=out_dir,
        config=LogisticRegressionConfig(seed=42, max_iter=100),
        groups=("low_level", "high_level"),
    )

    assert set(results.keys()) == {"logreg_low_level", "logreg_high_level"}
    assert (out_dir / "logreg_low_level" / "metrics.json").exists()
    assert (out_dir / "logreg_low_level" / "test_predictions.csv").exists()
    assert (out_dir / "logreg_high_level" / "metrics.json").exists()
    assert (out_dir / "logreg_high_level" / "test_predictions.csv").exists()


def test_cli_execution(tmp_path: Path):
    """Verify command line interface works cleanly."""
    data_file = tmp_path / "mock_higgs.csv.gz"
    rng = np.random.default_rng(101)
    labels = rng.integers(0, 2, size=(60, 1)).astype(np.float64)
    features = rng.normal(0.0, 1.0, size=(60, 28)).astype(np.float64)
    matrix = np.hstack([labels, features])

    with gzip.open(data_file, "wt", encoding="utf-8") as f:
        for row in matrix:
            f.write(",".join(f"{val:.6f}" for val in row) + "\n")

    out_dir = tmp_path / "cli_run"
    exit_code = main([
        "--data", str(data_file),
        "--rows", "50",
        "--feature-group", "low_level",
        "--output-dir", str(out_dir),
        "--seed", "42",
        "--max-iter", "100",
    ])
    assert exit_code == 0
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "test_predictions.csv").exists()
