"""Artifact discovery and validation loader for the HiggsLens dashboard.

Loads and validates experiment artifacts complying with the higgs-experiment-v1
schema under outputs/. Handles missing directories, malformed JSON, missing
prediction files, and invalid data types gracefully without crashing or faking data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from higgs_event_lab.protocol import ARTIFACT_SCHEMA_VERSION


class RunStatus(str, Enum):
    """Status classification for discovered experiment folders."""

    VALID = "valid"
    INCOMPLETE = "incomplete"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class IncompleteRunInfo:
    """Diagnostic details for an incomplete experiment run."""

    run_dir: str
    run_id: str
    reason: str
    missing_files: list[str]


@dataclass(frozen=True)
class MalformedRunInfo:
    """Diagnostic details for a malformed or schema-violating experiment run."""

    run_dir: str
    run_id: str
    reason: str
    error_details: str


@dataclass
class ExperimentArtifact:
    """Structured, validated experiment artifact for dashboard visualization."""

    run_id: str
    run_dir: Path
    model_name: str
    model_family: str
    feature_group: str
    feature_count: int
    feature_names: list[str]
    data_source: str
    row_count: int
    split: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    test_predictions: pd.DataFrame
    created_at_utc: str | None = None
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_mlp(self) -> bool:
        """Return True if this run corresponds to a neural network MLP."""
        return "mlp" in self.model_name.lower() or "neural" in self.model_family.lower()

    @property
    def is_logistic_regression(self) -> bool:
        """Return True if this run corresponds to Logistic Regression."""
        return "logreg" in self.model_name.lower() or "logistic" in self.model_name.lower()

    @property
    def has_training_history(self) -> bool:
        """Return True if genuine training epoch history is exported."""
        history = self.training.get("history")
        return bool(isinstance(history, list) and len(history) > 0)

    @property
    def training_history(self) -> list[dict[str, Any]]:
        """Return genuine training history list, or empty list if unavailable."""
        history = self.training.get("history")
        if isinstance(history, list):
            return history
        return []

    @property
    def training_seconds(self) -> float | None:
        """Return training runtime in seconds if recorded."""
        val = self.training.get("training_seconds")
        if isinstance(val, (int, float)):
            return float(val)
        return None

    @property
    def test_metrics(self) -> dict[str, Any]:
        """Return test evaluation metrics dictionary."""
        return self.evaluation.get("test", {})

    @property
    def validation_metrics(self) -> dict[str, Any]:
        """Return validation evaluation metrics dictionary."""
        return self.evaluation.get("validation", {})

    @property
    def test_roc_auc(self) -> float | None:
        """Return held-out test ROC-AUC score."""
        val = self.test_metrics.get("roc_auc")
        return float(val) if isinstance(val, (int, float)) else None

    @property
    def test_pr_auc(self) -> float | None:
        """Return held-out test PR-AUC (average precision) score."""
        val = self.test_metrics.get("pr_auc")
        return float(val) if isinstance(val, (int, float)) else None

    @property
    def test_brier_score(self) -> float | None:
        """Return held-out test Brier score loss."""
        val = self.test_metrics.get("brier_score")
        return float(val) if isinstance(val, (int, float)) else None

    @property
    def test_labels(self) -> np.ndarray:
        """Return held-out true class labels as integer array."""
        return self.test_predictions["class_label"].to_numpy(dtype=np.int64)

    @property
    def test_scores(self) -> np.ndarray:
        """Return held-out predicted classifier probability scores."""
        return self.test_predictions["score"].to_numpy(dtype=np.float64)

    @property
    def operating_point(self) -> dict[str, Any] | None:
        """Return validation-selected operating point metadata if exported."""
        op = self.evaluation.get("operating_point")
        return op if isinstance(op, dict) else None


@dataclass
class RunDiscoveryResult:
    """Summary of discovered, valid, incomplete, and malformed runs."""

    valid_runs: list[ExperimentArtifact] = field(default_factory=list)
    incomplete_runs: list[IncompleteRunInfo] = field(default_factory=list)
    malformed_runs: list[MalformedRunInfo] = field(default_factory=list)
    base_dir: str = "outputs"
    dir_exists: bool = True

    @property
    def total_scanned(self) -> int:
        """Return total number of folders scanned."""
        return len(self.valid_runs) + len(self.incomplete_runs) + len(self.malformed_runs)

    @property
    def has_valid_runs(self) -> bool:
        """Return True if at least one valid run is available."""
        return len(self.valid_runs) > 0


REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "model",
    "features",
    "data",
    "split",
    "training",
    "evaluation",
)

REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "row_index",
    "class_label",
    "score",
)


def validate_metrics_json(data: dict[str, Any]) -> str | None:
    """Validate metrics dictionary against higgs-experiment-v1 contract.

    Returns None if valid, or an error description string if invalid.
    """
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            return f"Missing required top-level key: '{key}'"

    schema_version = data.get("schema_version")
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        return (
            f"Unsupported schema_version '{schema_version}'. Expected '{ARTIFACT_SCHEMA_VERSION}'"
        )

    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return "run_id must be a non-empty string"

    model = data.get("model")
    if not isinstance(model, dict) or "name" not in model:
        return "model section must be a dictionary containing 'name'"

    features = data.get("features")
    if not isinstance(features, dict):
        return "features section must be a dictionary"
    if "group" not in features or features["group"] not in {"low_level", "high_level", "all"}:
        return "features.group must be 'low_level', 'high_level', or 'all'"
    if "count" not in features or not isinstance(features["count"], int) or features["count"] <= 0:
        return "features.count must be a positive integer"
    if "names" not in features or not isinstance(features["names"], list):
        return "features.names must be a list of feature column names"

    split = data.get("split")
    if not isinstance(split, dict):
        return "split section must be a dictionary"
    if "seed" not in split or not isinstance(split["seed"], int):
        return "split.seed must be an integer"

    eval_sec = data.get("evaluation")
    if not isinstance(eval_sec, dict):
        return "evaluation section must be a dictionary"
    if "test" not in eval_sec or not isinstance(eval_sec["test"], dict):
        return "evaluation must contain a 'test' metrics dictionary"

    test_metrics = eval_sec["test"]
    for m in ("roc_auc", "pr_auc", "brier_score"):
        if m in test_metrics:
            val = test_metrics[m]
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                return f"evaluation.test.{m} must be a finite numeric value"

    return None


def validate_test_predictions_df(df: pd.DataFrame) -> str | None:
    """Validate predictions DataFrame against schema contract.

    Returns None if valid, or an error description string if invalid.
    """
    if len(df) == 0:
        return "test_predictions.csv contains 0 rows (empty dataset)"

    missing_cols = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing_cols:
        return f"test_predictions.csv missing required columns: {missing_cols}"

    # Verify label values
    labels = df["class_label"]
    if not pd.api.types.is_numeric_dtype(labels):
        return "class_label column must be numeric"

    unique_labels = set(labels.dropna().unique())
    if not unique_labels.issubset({0, 1, 0.0, 1.0}):
        return f"class_label must only contain binary values {{0, 1}}, found {unique_labels}"

    # Verify score values
    scores = df["score"]
    if not pd.api.types.is_numeric_dtype(scores):
        return "score column must be numeric"

    score_arr = scores.to_numpy(dtype=np.float64)
    if not np.isfinite(score_arr).all():
        return "score column contains non-finite (NaN or Inf) values"

    if (score_arr < 0.0).any() or (score_arr > 1.0).any():
        return "score values must be bounded probabilities within [0.0, 1.0]"

    return None


def load_single_run(run_dir: Path) -> tuple[ExperimentArtifact | None, RunStatus, str | None, list[str]]:
    """Attempt to load and validate a single run directory.

    Returns: (artifact, status, reason, missing_files)
    """
    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "test_predictions.csv"

    missing = []
    if not metrics_path.is_file():
        missing.append("metrics.json")
    if not preds_path.is_file():
        missing.append("test_predictions.csv")

    if missing:
        return None, RunStatus.INCOMPLETE, f"Missing required file(s): {', '.join(missing)}", missing

    # Parse JSON
    try:
        metrics_text = metrics_path.read_text(encoding="utf-8")
        raw_metrics = json.loads(metrics_text)
    except Exception as exc:
        return None, RunStatus.MALFORMED, f"Unparseable metrics.json: {exc}", []

    if not isinstance(raw_metrics, dict):
        return None, RunStatus.MALFORMED, "metrics.json must contain a top-level JSON object", []

    schema_err = validate_metrics_json(raw_metrics)
    if schema_err:
        return None, RunStatus.MALFORMED, f"Schema validation error: {schema_err}", []

    # Parse CSV
    try:
        df_preds = pd.read_csv(preds_path)
    except Exception as exc:
        return None, RunStatus.MALFORMED, f"Unparseable test_predictions.csv: {exc}", []

    csv_err = validate_test_predictions_df(df_preds)
    if csv_err:
        return None, RunStatus.MALFORMED, f"Prediction file validation error: {csv_err}", []

    artifact = ExperimentArtifact(
        run_id=str(raw_metrics["run_id"]),
        run_dir=run_dir,
        model_name=str(raw_metrics["model"]["name"]),
        model_family=str(raw_metrics["model"].get("family", "unknown")),
        feature_group=str(raw_metrics["features"]["group"]),
        feature_count=int(raw_metrics["features"]["count"]),
        feature_names=list(raw_metrics["features"]["names"]),
        data_source=str(raw_metrics["data"].get("source", "unknown")),
        row_count=int(raw_metrics["data"].get("row_count", 0)),
        split=dict(raw_metrics["split"]),
        training=dict(raw_metrics.get("training", {})),
        evaluation=dict(raw_metrics.get("evaluation", {})),
        test_predictions=df_preds,
        created_at_utc=raw_metrics.get("created_at_utc"),
        raw_metrics=raw_metrics,
    )

    return artifact, RunStatus.VALID, None, []


def discover_and_load_runs(base_dir: str | Path = "outputs") -> RunDiscoveryResult:
    """Discover and parse experiment runs beneath base_dir.

    Never crashes if base_dir does not exist or contains invalid files.
    """
    path = Path(base_dir)
    result = RunDiscoveryResult(base_dir=str(path))

    if not path.exists():
        result.dir_exists = False
        return result

    if not path.is_dir():
        result.dir_exists = False
        result.malformed_runs.append(
            MalformedRunInfo(
                run_dir=str(path),
                run_id="root",
                reason=f"Path '{path}' is not a directory",
                error_details="Path is a regular file or link",
            )
        )
        return result

    result.dir_exists = True

    # Scan for candidate experiment directories (immediate subdirectories)
    # Also check if base_dir itself contains metrics.json and test_predictions.csv
    candidate_dirs: list[Path] = []

    if (path / "metrics.json").is_file() or (path / "test_predictions.csv").is_file():
        candidate_dirs.append(path)

    for item in sorted(path.iterdir()):
        if item.is_dir() and not item.name.startswith((".", "__")):
            if item.name in {"curves", "runs", "model_weights", "eda"} and not (item / "metrics.json").is_file():
                continue
            candidate_dirs.append(item)

    for run_dir in candidate_dirs:
        artifact, status, reason, missing = load_single_run(run_dir)
        folder_name = run_dir.name

        if status == RunStatus.VALID and artifact is not None:
            result.valid_runs.append(artifact)
        elif status == RunStatus.INCOMPLETE:
            result.incomplete_runs.append(
                IncompleteRunInfo(
                    run_dir=str(run_dir),
                    run_id=folder_name,
                    reason=reason or "Incomplete run",
                    missing_files=missing,
                )
            )
        elif status == RunStatus.MALFORMED:
            result.malformed_runs.append(
                MalformedRunInfo(
                    run_dir=str(run_dir),
                    run_id=folder_name,
                    reason=reason or "Malformed run",
                    error_details=reason or "Schema or data constraint violated",
                )
            )

    return result
