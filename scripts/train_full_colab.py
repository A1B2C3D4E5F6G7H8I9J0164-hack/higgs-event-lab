"""Tuned full-scale benchmark run for Colab or Kaggle.

The notebook cells download the full UCI archive and prepare a processed
subset; this script then trains the three mid-semester baselines with stronger
settings, evaluates once on the shared held-out split, and writes a single
summary plus per-model test predictions for the report and dashboard.

Usage:
    HIGGS_SAMPLE_PATH=data/processed/higgs_sample_1m.csv.gz \
        python scripts/train_full_colab.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUTS_DIR = ROOT_DIR / "outputs"

from higgs_event_lab.evaluation import compute_classification_metrics, select_operating_threshold  # noqa: E402
from higgs_event_lab.features import ALL_COLUMNS, ALL_FEATURES, TARGET_COLUMN  # noqa: E402
from higgs_event_lab.neural import MLPConfig, train_mlp  # noqa: E402
from higgs_event_lab.protocol import SplitConfig, create_stratified_splits  # noqa: E402

PRIMARY_SAMPLE = ROOT_DIR / "data" / "processed" / "higgs_sample_1m.csv.gz"
FALLBACK_SAMPLE = ROOT_DIR / "data" / "processed" / "higgs_sample_250k.csv.gz"


def resolve_sample_path(requested: str | None) -> Path:
    """Prefer an explicit path, then the large processed sample, then the 250k one."""
    for candidate in (requested, os.environ.get("HIGGS_SAMPLE_PATH"), str(PRIMARY_SAMPLE), str(FALLBACK_SAMPLE)):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "No processed HIGGS sample found. Run the notebook download and subset cells first."
    )


def _relative_path(path: Path) -> str:
    """Represent a path relative to the repository when possible."""
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def evaluate_run(
    name: str,
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    seconds: float,
    *,
    training: dict[str, object] | None = None,
) -> dict[str, object]:
    """Compute consistent validation, test and frozen operating-point metrics."""
    threshold = select_operating_threshold(validation_labels, validation_scores, 0.50)
    test_signal = test_labels == 1
    test_background = test_labels == 0
    validation_signal = validation_labels == 1
    validation_background = validation_labels == 0
    validation_metrics = compute_classification_metrics(validation_labels, validation_scores)
    test_metrics = compute_classification_metrics(test_labels, test_scores)
    test_metrics["accuracy"] = float(np.mean((test_scores >= 0.5) == test_labels))
    validation_metrics["accuracy"] = float(np.mean((validation_scores >= 0.5) == validation_labels))
    operating_point = {
        "target_signal_efficiency": 0.50,
        "validation_threshold": float(threshold),
        "validation_actual_signal_efficiency": float(np.mean(validation_scores[validation_signal] >= threshold)),
        "validation_background_rejection": float(np.mean(validation_scores[validation_background] < threshold)),
        "test_actual_signal_efficiency": float(np.mean(test_scores[test_signal] >= threshold)),
        "test_background_rejection": float(np.mean(test_scores[test_background] < threshold)),
    }
    record: dict[str, object] = {
        "model": name,
        "feature_group": "all",
        "feature_names": list(ALL_FEATURES),
        "validation": validation_metrics,
        "test": test_metrics,
        "operating_point": operating_point,
        "test_roc_auc": round(float(test_metrics["roc_auc"]), 6),
        "test_pr_auc": round(float(test_metrics["pr_auc"]), 6),
        "brier_score": round(float(test_metrics["brier_score"]), 6),
        "test_accuracy": round(float(test_metrics["accuracy"]), 6),
        "training_seconds": round(seconds, 2),
    }
    if training:
        record["training"] = training
    return record


def train_and_score_all(sample_path: Path, output_dir: Path = OUTPUTS_DIR) -> tuple[list[dict[str, object]], list[Path]]:
    """Run the full benchmark once; the split stays frozen across every model."""
    import joblib
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(sample_path, header=None, names=list(ALL_COLUMNS))
    print(f"Loaded {len(df):,} rows from {sample_path.name}")

    labels = df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    features = df.drop(columns=[TARGET_COLUMN])
    split = create_stratified_splits(labels, SplitConfig(seed=42))
    y_train, y_val, y_test = labels[split.train], labels[split.validation], labels[split.test]
    print(f"Split sizes -> train {len(y_train):,}, validation {len(y_val):,}, test {len(y_test):,}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features.iloc[split.train].to_numpy(dtype=np.float64))
    X_val = scaler.transform(features.iloc[split.validation].to_numpy(dtype=np.float64))
    X_test = scaler.transform(features.iloc[split.test].to_numpy(dtype=np.float64))

    # Stronger logistic regression: more iterations on the standardised inputs.
    started = time.perf_counter()
    logreg = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    logreg.fit(X_train, y_train)
    logreg_seconds = time.perf_counter() - started
    logreg_validation_scores = logreg.predict_proba(X_val)[:, 1]
    logreg_scores = logreg.predict_proba(X_test)[:, 1]

    started = time.perf_counter()
    xgb_module = __import__("xgboost")
    xgboost = xgb_module.XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    xgboost.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_seconds = time.perf_counter() - started
    xgb_validation_scores = xgboost.predict_proba(X_val)[:, 1]
    xgb_scores = xgboost.predict_proba(X_test)[:, 1]

    # Wider compact MLP: same batch norm / dropout recipe, more capacity and epochs.
    started = time.perf_counter()
    mlp_config = MLPConfig(
        hidden_sizes=(256, 128, 64),
        dropout=0.30,
        learning_rate=8e-4,
        weight_decay=2e-5,
        max_epochs=60,
        early_stopping_patience=5,
        batch_size=1024,
        seed=42,
    )
    mlp_result = train_mlp(features, labels, split=split, config=mlp_config)
    mlp_seconds = time.perf_counter() - started
    mlp_validation_scores = np.asarray(mlp_result["validation"]["probabilities"], dtype=np.float64)
    mlp_scores = np.asarray(mlp_result["test"]["probabilities"], dtype=np.float64)

    records = [
        evaluate_run(
            "logistic_regression_tuned", y_val, logreg_validation_scores, y_test, logreg_scores, logreg_seconds,
            training={"solver": "lbfgs", "C": 1.0, "max_iter": 2000},
        ),
        evaluate_run(
            "xgboost_tuned", y_val, xgb_validation_scores, y_test, xgb_scores, xgb_seconds,
            training={"n_estimators": 600, "max_depth": 7, "learning_rate": 0.06, "subsample": 0.9, "colsample_bytree": 0.9, "tree_method": "hist"},
        ),
        evaluate_run(
            "wide_mlp_tuned", y_val, mlp_validation_scores, y_test, mlp_scores, mlp_seconds,
            training={"config": {key: list(value) if isinstance(value, tuple) else value for key, value in mlp_config.__dict__.items()}, "device": mlp_result["device"], "epochs_completed": len(mlp_result["history"]), "history": mlp_result["history"]},
        ),
    ]

    prediction_sources = {
        "logistic_regression_tuned": logreg_scores,
        "xgboost_tuned": xgb_scores,
        "wide_mlp_tuned": mlp_scores,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for run_name, scores in prediction_sources.items():
        prediction_path = output_dir / f"{run_name}_test_predictions.csv"
        pd.DataFrame({"row_index": split.test, "class_label": y_test, "score": scores}).to_csv(
            prediction_path, index=False
        )

    # Persist trained weights so the report can load them again without retraining.
    model_dir = output_dir / "model_weights"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, model_dir / "logistic_regression_tuned.joblib")
    joblib.dump(xgboost, model_dir / "xgboost_tuned.joblib")
    joblib.dump(scaler, model_dir / "feature_scaler.joblib")
    torch.save(mlp_result["model"], model_dir / "wide_mlp_tuned.pt")

    summary_path = output_dir / "full_colab_results.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "higgs-experiment-v1",
                "run_kind": "full_colab_benchmark",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sample_path": _relative_path(sample_path),
                "rows": len(df),
                "feature_group": "all",
                "feature_names": list(ALL_FEATURES),
                "split": {"seed": 42, "validation_fraction": 0.15, "test_fraction": 0.15, "train_rows": len(y_train), "validation_rows": len(y_val), "test_rows": len(y_test)},
                "runs": records,
                "artifacts": [
                    _relative_path(output_dir / f"{name}_test_predictions.csv")
                    for name in prediction_sources
                ]
                + [_relative_path(p) for p in sorted(model_dir.glob("*"))],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Summary written to", _relative_path(summary_path))
    return records, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train tuned mid-semester baselines for the Colab run.")
    parser.add_argument("--sample", default=None, help="Processed sample CSV; falls back to the 1m then 250k sample.")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR), help="Directory for summary, predictions and weights.")
    args = parser.parse_args(argv)

    sample_path = resolve_sample_path(args.sample)
    output_dir = Path(args.output_dir)
    train_and_score_all(sample_path, output_dir=output_dir)

    print("\nResults (frozen test split):")
    # Recompute a short summary line by reading back the saved JSON.
    summary = json.loads((output_dir / "full_colab_results.json").read_text(encoding="utf-8"))
    for record in summary["runs"]:
        print(
            f"  {record['model']:<28} ROC-AUC {record['test_roc_auc']:.4f}"
            f"  PR-AUC {record['test_pr_auc']:.4f}  Brier {record['brier_score']:.4f}"
            f"  accuracy {record['test_accuracy']:.4f}  seconds {record['training_seconds']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
