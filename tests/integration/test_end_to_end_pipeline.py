"""End-to-end integration test connecting data loading, protocol splits, and MLP training."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from higgs_event_lab.data import load_higgs_sample
from higgs_event_lab.features import HIGH_LEVEL_FEATURES, LOW_LEVEL_FEATURES
from higgs_event_lab.neural import MLPConfig, main, train_mlp, write_run_artifacts
from higgs_event_lab.protocol import (
    ARTIFACT_SCHEMA_VERSION,
    SplitConfig,
    create_stratified_splits,
)


@pytest.fixture
def synthetic_higgs_dataset(tmp_path: Path) -> Path:
    """Generate a valid synthetic HIGGS gzipped CSV with 120 events and 28 features."""
    file_path = tmp_path / "synthetic_higgs_sample.csv.gz"
    rng = np.random.default_rng(42)
    n_rows = 120
    labels = rng.integers(0, 2, size=(n_rows, 1)).astype(np.float64)
    features = rng.normal(0.0, 1.0, size=(n_rows, 28)).astype(np.float64)
    matrix = np.hstack([labels, features])

    with gzip.open(file_path, "wt", encoding="utf-8") as handle:
        for row in matrix:
            handle.write(",".join(f"{val:.6f}" for val in row) + "\n")
    return file_path


def test_end_to_end_data_to_mlp_artifacts(synthetic_higgs_dataset: Path, tmp_path: Path):
    """Verify that data pipeline, protocol splits, and MLP baseline work together cleanly."""
    # 1. Load data via chunked sample loader
    batch = load_higgs_sample(synthetic_higgs_dataset, n_rows=100, feature_group="all")
    assert batch.X.shape == (100, 28)
    assert batch.y.shape == (100,)

    # 2. Split according to shared deterministic experiment protocol
    split = create_stratified_splits(batch.y, SplitConfig(seed=42))
    assert len(split.train) + len(split.validation) + len(split.test) == 100
    assert not (set(split.train) & set(split.validation))
    assert not (set(split.train) & set(split.test))

    # 3. Train compact MLP binary classifier
    config = MLPConfig(
        hidden_sizes=(16, 8),
        dropout=0.1,
        batch_size=32,
        max_epochs=4,
        early_stopping_patience=2,
        device="cpu",
        seed=42,
    )
    result = train_mlp(batch.X, batch.y, split=split, config=config)

    assert "model" in result
    assert "validation" in result
    assert "test" in result
    assert result["parameter_count"] > 0
    assert len(result["history"]) >= 1

    # 4. Export run artifacts
    run_dir = tmp_path / "experiment_run_all"
    metadata_path, predictions_path = write_run_artifacts(
        result,
        run_dir=run_dir,
        run_id="test_e2e_run",
        feature_group="all",
        feature_names=list(batch.X.columns),
        data_source=str(synthetic_higgs_dataset),
        row_count=len(batch.X),
        config=config,
    )

    assert metadata_path.exists()
    assert predictions_path.exists()

    # 5. Verify metrics.json complies with higgs-experiment-v1 contract
    envelope = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert envelope["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert envelope["run_id"] == "test_e2e_run"
    assert envelope["model"]["name"] == "compact_mlp"
    assert envelope["features"]["group"] == "all"
    assert envelope["features"]["count"] == 28
    assert envelope["split"]["seed"] == 42
    assert envelope["split"]["train_rows"] == len(split.train)
    assert envelope["split"]["validation_rows"] == len(split.validation)
    assert envelope["split"]["test_rows"] == len(split.test)

    for split_key in ("validation", "test"):
        metrics = envelope["evaluation"][split_key]
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert len(metrics["calibration"]["fraction_positives"]) == len(metrics["calibration"]["mean_predicted_value"])

    # 6. Verify test_predictions.csv
    pred_df = pd.read_csv(predictions_path)
    assert list(pred_df.columns) == ["row_index", "class_label", "score"]
    assert len(pred_df) == len(split.test)
    assert (pred_df["score"] >= 0.0).all() and (pred_df["score"] <= 1.0).all()


def test_cli_invocation_with_feature_group(synthetic_higgs_dataset: Path, tmp_path: Path):
    """Verify CLI entrypoint works with low_level feature group."""
    output_dir = tmp_path / "mlp_low_level_run"
    exit_code = main([
        "--data", str(synthetic_higgs_dataset),
        "--rows", "80",
        "--feature-group", "low_level",
        "--output-dir", str(output_dir),
        "--seed", "123",
        "--epochs", "2",
    ])
    assert exit_code == 0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "test_predictions.csv").exists()

    metadata = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metadata["features"]["group"] == "low_level"
    assert metadata["features"]["count"] == len(LOW_LEVEL_FEATURES)
    assert metadata["features"]["names"] == list(LOW_LEVEL_FEATURES)
