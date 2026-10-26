"""Tuned full-scale benchmark run for Colab or Kaggle.

The notebook cells download the full UCI archive and prepare a processed
subset; this script then trains the three mid-semester baselines with stronger
settings, evaluates once on the shared held-out split, and writes a single
summary plus per-model test predictions for the report and dashboard.

Usage:
    HIGGS_SAMPLE_PATH=data/processed/higgs_sample_1m.csv.gz \
        python scripts/train_full_colab.py --output-dir outputs/full_colab
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

from higgs_event_lab.features import ALL_COLUMNS, TARGET_COLUMN  # noqa: E402
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


def evaluate_run(name: str, labels: np.ndarray, scores: np.ndarray, seconds: float) -> dict[str, float]:
    """Compute the four headline metrics that go straight into the report."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    preds = (scores >= 0.5).astype(int)
    return {
        "model": name,
        "test_roc_auc": round(float(roc_auc_score(labels, scores)), 6),
        "test_pr_auc": round(float(average_precision_score(labels, scores)), 6),
        "brier_score": round(float(brier_score_loss(labels, scores)), 6),
        "test_accuracy": round(float((preds == labels).mean()), 6),
        "training_seconds": round(seconds, 2),
    }


def train_and_score_all(sample_path: Path) -> tuple[list[dict[str, float]], list[Path]]:
    """Run the full benchmark once; the split stays frozen across every model."""
    import joblib
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
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
    seconds = time.perf_counter() - started
    logreg_scores = logreg.predict_proba(X_test)[:, 1]

    # Tuned XGBoost: deeper trees and a slower learning rate than the quick baseline.
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
    xgboost.fit(X_train, y_train)
    seconds = time.perf_counter() - started
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
    seconds = time.perf_counter() - started
    mlp_scores = np.asarray(mlp_result["test"]["probabilities"], dtype=np.float64)

    records = [
        evaluate_run("logistic_regression_tuned", y_test, logreg_scores, seconds),
        evaluate_run("xgboost_tuned", y_test, xgb_scores, seconds),
        evaluate_run("wide_mlp_tuned", y_test, mlp_scores, seconds),
    ]

    prediction_sources = {
        "logistic_regression_tuned": logreg_scores,
        "xgboost_tuned": xgb_scores,
        "wide_mlp_tuned": mlp_scores,
    }
    for run_name, scores in prediction_sources.items():
        prediction_path = OUTPUTS_DIR / f"{run_name}_test_predictions.csv"
        pd.DataFrame({"row_index": split.test, "class_label": y_test, "score": scores}).to_csv(
            prediction_path, index=False
        )

    # Persist trained weights so the report can load them again without retraining.
    model_dir = OUTPUTS_DIR / "model_weights"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, model_dir / "logistic_regression_tuned.joblib")
    joblib.dump(xgboost, model_dir / "xgboost_tuned.joblib")
    joblib.dump(scaler, model_dir / "feature_scaler.joblib")
    torch.save(mlp_result["model"], model_dir / "wide_mlp_tuned.pt")

    summary_path = OUTPUTS_DIR / "full_colab_results.json"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "higgs-experiment-v1",
                "run_kind": "full_colab_benchmark",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sample_path": str(sample_path.relative_to(ROOT_DIR)),
                "rows": len(df),
                "split": {"seed": 42, "validation_fraction": 0.15, "test_fraction": 0.15},
                "runs": records,
                "artifacts": [
                    str((OUTPUTS_DIR / f"{name}_test_predictions.csv").relative_to(ROOT_DIR))
                    for name in prediction_sources
                ]
                + [str(p.relative_to(ROOT_DIR)) for p in sorted(model_dir.glob("*"))],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Summary written to", summary_path.relative_to(ROOT_DIR))
    return records, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train tuned mid-semester baselines for the Colab run.")
    parser.add_argument("--sample", default=None, help="Processed sample CSV; falls back to the 1m then 250k sample.")
    args = parser.parse_args(argv)

    sample_path = resolve_sample_path(args.sample)
    train_and_score_all(sample_path)

    print("\nResults (frozen test split):")
    # Recompute a short summary line by reading back the saved JSON.
    summary = json.loads((OUTPUTS_DIR / "full_colab_results.json").read_text(encoding="utf-8"))
    for record in summary["runs"]:
        print(
            f"  {record['model']:<28} ROC-AUC {record['test_roc_auc']:.4f}"
            f"  PR-AUC {record['test_pr_auc']:.4f}  Brier {record['brier_score']:.4f}"
            f"  accuracy {record['test_accuracy']:.4f}  seconds {record['training_seconds']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
