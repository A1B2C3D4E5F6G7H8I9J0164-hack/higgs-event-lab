"""Unit tests for the dashboard view components and sample artifact generator."""

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.loader import discover_and_load_runs
from app.utils.loader import inspect_pipeline_status, load_experiments
from scripts.generate_sample_artifacts import main as generate_artifacts


def test_generate_sample_artifacts_produces_valid_schema(tmp_path, monkeypatch):
    """Verify generate_sample_artifacts populates valid higgs-experiment-v1 artifacts."""
    monkeypatch.setattr("scripts.generate_sample_artifacts.OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr("scripts.generate_sample_artifacts.CURVES_DIR", tmp_path / "outputs" / "curves")

    generate_artifacts()

    outputs_dir = tmp_path / "outputs"
    assert outputs_dir.is_dir()
    assert (outputs_dir / "experiments.json").is_file()

    # Verify discovery under app.loader
    discovery = discover_and_load_runs(outputs_dir)
    assert len(discovery.valid_runs) == 9
    assert len(discovery.incomplete_runs) == 0
    assert len(discovery.malformed_runs) == 0

    # Verify discovery under app.utils.loader
    loaded = load_experiments(tmp_path)
    assert len(loaded.experiments) >= 9
    assert len(loaded.curves) >= 9
    assert not loaded.errors


def test_bootstrap_cloud_artifacts_if_needed(tmp_path, monkeypatch):
    """Verify that cloud entrypoint bootstrap creates runs when outputs/ is empty."""
    import streamlit_app

    fake_outputs = tmp_path / "outputs"
    fake_outputs.mkdir()
    monkeypatch.setattr(streamlit_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr("scripts.generate_sample_artifacts.OUTPUTS_DIR", fake_outputs)
    monkeypatch.setattr("scripts.generate_sample_artifacts.CURVES_DIR", fake_outputs / "curves")

    streamlit_app.bootstrap_cloud_artifacts_if_needed()

    runs = [d for d in fake_outputs.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    assert len(runs) == 9
