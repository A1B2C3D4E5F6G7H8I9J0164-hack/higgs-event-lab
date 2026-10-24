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

from higgs_event_lab.neural import MLPConfig, build_mlp, train_mlp, write_run_artifacts


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
