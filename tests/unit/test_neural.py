"""Focused shape, reproducibility, and artifact tests for Rachit's MLP module."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from higgs_event_lab.neural import (
    MLPConfig,
    build_mlp,
    run_feature_group_benchmarks,
    train_mlp,
    write_run_artifacts,
)


def test_mlp_has_binary_logit_and_batch_norm_dropout():
    model = build_mlp(7, MLPConfig(hidden_sizes=(8,), dropout=0.25))
    import torch
    assert model(torch.zeros(3, 7)).shape == (3, 1)
    assert any(isinstance(layer, torch.nn.BatchNorm1d) for layer in model)
    assert any(isinstance(layer, torch.nn.Dropout) for layer in model)


def test_tiny_mlp_run_exports_measured_test_predictions(tmp_path: Path):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(np.int64)
    result = train_mlp(X, y, config=MLPConfig(hidden_sizes=(8,), batch_size=16, max_epochs=3, early_stopping_patience=2, device="cpu", seed=4))
    assert set(result["test"]["metrics"]) == {"roc_auc", "pr_auc", "brier_score", "calibration"}
    metadata, predictions = write_run_artifacts(result, run_dir=tmp_path / "run", run_id="synthetic", feature_group="all", feature_names=["a", "b", "c", "d"], data_source="synthetic-test", row_count=len(X), config=MLPConfig(hidden_sizes=(8,), device="cpu", seed=4))
    assert metadata.exists() and predictions.exists()


def test_run_feature_group_benchmarks(tmp_path: Path):
    import gzip
    data_file = tmp_path / "test_higgs.csv.gz"
    rng = np.random.default_rng(123)
    labels = rng.integers(0, 2, size=(60, 1)).astype(np.float64)
    features = rng.normal(0, 1, size=(60, 28)).astype(np.float64)
    matrix = np.hstack([labels, features])
    with gzip.open(data_file, "wt", encoding="utf-8") as handle:
        for row in matrix:
            handle.write(",".join(f"{v:.6f}" for v in row) + "\n")

    out_dir = tmp_path / "benchmarks"
    results = run_feature_group_benchmarks(
        data_path=data_file,
        row_count=50,
        output_base_dir=out_dir,
        config=MLPConfig(hidden_sizes=(8,), batch_size=16, max_epochs=2, early_stopping_patience=2, device="cpu", seed=1),
        groups=("low_level", "high_level"),
    )
    assert set(results.keys()) == {"low_level", "high_level"}
    assert (out_dir / "mlp_low_level" / "metrics.json").exists()
    assert (out_dir / "mlp_high_level" / "metrics.json").exists()
