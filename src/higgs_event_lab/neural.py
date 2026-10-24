"""Compact, reproducible PyTorch MLP for simulated HIGGS event classification.

This module deliberately keeps data loading, splitting, preprocessing, training,
and artifact writing explicit.  It does not fabricate results and never trains
when imported by the dashboard.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from higgs_event_lab.data import load_higgs_sample
from higgs_event_lab.evaluation import compute_classification_metrics
from higgs_event_lab.protocol import SplitConfig, SplitIndices, create_stratified_splits, new_experiment_artifact


def _torch() -> Any:
    """Import torch lazily so data-only users do not need a GPU stack installed."""
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for MLP training. Install requirements.txt first.") from exc
    return torch


@dataclass(frozen=True)
class MLPConfig:
    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 30
    early_stopping_patience: int = 5
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(width <= 0 for width in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer widths")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if min(self.learning_rate, self.weight_decay) < 0:
            raise ValueError("learning_rate and weight_decay must be non-negative")
        if min(self.batch_size, self.max_epochs, self.early_stopping_patience) <= 0:
            raise ValueError("batch_size, max_epochs, and patience must be positive")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch before a run."""
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> str:
    torch = _torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    return requested


def build_mlp(input_features: int, config: MLPConfig) -> Any:
    """Build a compact binary classifier with BatchNorm and Dropout in each block."""
    if input_features <= 0:
        raise ValueError("input_features must be positive")
    torch = _torch()
    layers: list[Any] = []
    previous = input_features
    for width in config.hidden_sizes:
        layers.extend((torch.nn.Linear(previous, width), torch.nn.BatchNorm1d(width), torch.nn.ReLU(), torch.nn.Dropout(config.dropout)))
        previous = width
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


def _as_float_matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    array = X.to_numpy(dtype=np.float32, copy=True) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 10 or array.shape[1] == 0:
        raise ValueError("X must be a two-dimensional matrix with at least 10 rows and one feature")
    if not np.isfinite(array).all():
        raise ValueError("X must contain only finite values")
    return array


def _predict(model: Any, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    torch = _torch()
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            probabilities.append(torch.sigmoid(model(batch)).squeeze(1).cpu().numpy())
    return np.concatenate(probabilities)


def _metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return compute_classification_metrics(y_true, probability)


def train_mlp(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | list[int],
    *,
    split: SplitIndices | None = None,
    config: MLPConfig = MLPConfig(),
) -> dict[str, Any]:
    """Train on the training split, select epoch by validation ROC-AUC, and test once.

    The fitted scaler is intentionally returned for local persistence but is not
    serialized automatically; callers must keep it with the model checkpoint.
    """
    x_array = _as_float_matrix(X)
    y_array = np.asarray(y, dtype=np.int64)
    if y_array.shape != (x_array.shape[0],):
        raise ValueError("y must be one-dimensional and have one label per X row")
    split = split or create_stratified_splits(y_array, SplitConfig(seed=config.seed))
    seed_everything(config.seed)
    device = resolve_device(config.device)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_array[split.train]).astype(np.float32)
    x_validation = scaler.transform(x_array[split.validation]).astype(np.float32)
    x_test = scaler.transform(x_array[split.test]).astype(np.float32)
    y_train, y_validation, y_test = y_array[split.train], y_array[split.validation], y_array[split.test]

    torch = _torch()
    model = build_mlp(x_array.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32)))
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)

    history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_validation_auc = -np.inf
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses: list[float] = []
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device)).squeeze(1)
            loss = loss_fn(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation_probability = _predict(model, x_validation, device, config.batch_size)
        validation_metrics = _metrics(y_validation, validation_probability)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_roc_auc": validation_metrics["roc_auc"]})
        if validation_metrics["roc_auc"] > best_validation_auc:
            best_validation_auc = validation_metrics["roc_auc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    training_seconds = time.perf_counter() - started
    assert best_state is not None
    model.load_state_dict(best_state)

    validation_probability = _predict(model, x_validation, device, config.batch_size)
    test_probability = _predict(model, x_test, device, config.batch_size)
    return {
        "model": model,
        "scaler": scaler,
        "split": split,
        "history": history,
        "device": device,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_seconds": training_seconds,
        "validation": {"row_indices": split.validation, "labels": y_validation, "probabilities": validation_probability, "metrics": _metrics(y_validation, validation_probability)},
        "test": {"row_indices": split.test, "labels": y_test, "probabilities": test_probability, "metrics": _metrics(y_test, test_probability)},
    }


def write_run_artifacts(
    result: dict[str, Any], *, run_dir: str | Path, run_id: str, feature_group: str,
    feature_names: list[str], data_source: str, row_count: int, config: MLPConfig,
) -> tuple[Path, Path]:
    """Write JSON metadata and test predictions for the dashboard; never fake values."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=False)
    artifact = new_experiment_artifact(run_id=run_id, model_name="compact_mlp", feature_group=feature_group, feature_names=feature_names, split=result["split"], data_source=data_source, row_count=row_count)
    artifact["training"] = {
        "config": asdict(config), "device": result["device"], "parameter_count": result["parameter_count"],
        "training_seconds": result["training_seconds"], "epochs_completed": len(result["history"]), "history": result["history"],
    }
    artifact["evaluation"] = {"validation": result["validation"]["metrics"], "test": result["test"]["metrics"]}
    artifact["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path = run_path / "metrics.json"
    metadata_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    prediction_path = run_path / "test_predictions.csv"
    pd.DataFrame({"row_index": result["test"]["row_indices"], "class_label": result["test"]["labels"], "score": result["test"]["probabilities"]}).to_csv(prediction_path, index=False)
    return metadata_path, prediction_path


def run_feature_group_benchmarks(
    *,
    data_path: str | Path,
    row_count: int,
    output_base_dir: str | Path = "outputs",
    config: MLPConfig = MLPConfig(),
    groups: tuple[str, ...] = ("low_level", "high_level", "all"),
) -> dict[str, dict[str, Any]]:
    """Run fair benchmarks across specified feature groups using identical split seed."""
    base = Path(output_base_dir)
    results: dict[str, dict[str, Any]] = {}
    for group in groups:
        batch = load_higgs_sample(data_path, n_rows=row_count, feature_group=group)
        split = create_stratified_splits(batch.y, SplitConfig(seed=config.seed))
        result = train_mlp(batch.X, batch.y, split=split, config=config)
        run_dir = base / f"mlp_{group}"
        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
        write_run_artifacts(
            result,
            run_dir=run_dir,
            run_id=f"mlp_{group}",
            feature_group=group,
            feature_names=list(batch.X.columns),
            data_source=str(data_path),
            row_count=row_count,
            config=config,
        )
        results[group] = result
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the compact HIGGS MLP on a documented subset.")
    parser.add_argument("--data", default="data/processed/higgs_sample_250k.csv.gz")
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--feature-group", choices=("low_level", "high_level", "all"), default="all")
    parser.add_argument("--all-groups", action="store_true", help="Train and export benchmarks for all 3 feature groups")
    parser.add_argument("--output-dir", default="outputs/mlp_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args(argv)

    config = MLPConfig(seed=args.seed, max_epochs=args.epochs)
    if args.all_groups:
        results = run_feature_group_benchmarks(
            data_path=args.data,
            row_count=args.rows,
            output_base_dir=args.output_dir,
            config=config,
        )
        print(f"Completed multi-group MLP benchmark across {list(results.keys())}")
        return 0

    batch = load_higgs_sample(args.data, n_rows=args.rows, feature_group=args.feature_group)
    result = train_mlp(batch.X, batch.y, config=config)
    metadata, predictions = write_run_artifacts(result, run_dir=args.output_dir, run_id=Path(args.output_dir).name, feature_group=args.feature_group, feature_names=list(batch.X.columns), data_source=args.data, row_count=len(batch.X), config=config)
    print(f"Wrote measured MLP artifacts: {metadata} and {predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
