"""Regression tests for safe dashboard model loading."""

from pathlib import Path

from app.views import simulation


def test_xgboost_weight_is_disabled_by_default(tmp_path: Path, monkeypatch):
    """A stale serialized XGBoost binary must not be deserialized during page load."""
    monkeypatch.setattr(simulation, "WEIGHTS_DIR", tmp_path)
    (tmp_path / "xgboost_tuned.joblib").write_bytes(b"incompatible-binary")
    monkeypatch.delenv("HIGGSLENS_ENABLE_XGB_WEIGHTS", raising=False)

    models, scaler = simulation.load_models()

    assert models == {}
    assert scaler is None
    assert any("XGBoost interactive weights are disabled" in note for note in simulation.MODEL_LOAD_NOTICES)
